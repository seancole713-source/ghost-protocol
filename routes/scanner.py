"""Routes: scanner — extracted from wolf_app.py (Step 12)"""
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

router = APIRouter()
LOGGER = logging.getLogger("ghost")

# --- 8 endpoints ---

@router.get("/api/scan/movers")
async def api_scan_movers():
    """
    Get real-time market movers for crypto and stocks.

    Returns:
        {
            "crypto": [...],
            "stocks": [...],
            "ts": int,
            "crypto_count": int,
            "stocks_count": int
        }
    """
    try:
        # Import movers scanner
        from app.core import movers_scanner

        # Create price fetch wrapper for the scanner
        async def fetch_price_wrapper(symbol: str, is_crypto: bool = False):
            """Wrapper to use existing fetch_price_live function"""
            try:
                if is_crypto:
                    # Use existing crypto price endpoint logic
                    result = await api_crypto_price(symbol)
                    return result
                else:
                    # Use existing stock price fetch
                    result = await fetch_price_live(symbol)
                    return {
                        "price": result[0] if result else None,
                        "provider": result[2] if result and len(result) > 2 else "unknown",
                        "ts": int(time.time() * 1000)
                    }
            except Exception as e:
                LOGGER.debug(f"Price fetch error for {symbol}: {e}")
                return None

        # Run scans with timeout
        try:
            crypto_task = movers_scanner.scan_crypto(
                fetch_price_wrapper,
                None,  # ohlcv_func not implemented yet
                REDIS
            )
            stocks_task = movers_scanner.scan_stocks(
                fetch_price_wrapper,
                None,  # ohlcv_func not implemented yet
                REDIS
            )

            # Execute with timeout
            crypto_movers, stock_movers = await asyncio.wait_for(
                asyncio.gather(crypto_task, stocks_task),
                timeout=movers_scanner.SCAN_TIMEOUT
            )
        except TimeoutError:
            return JSONResponse(
                {"ok": False, "detail": "Scan timeout exceeded"},
                status_code=504
            )

        # Build payload
        payload = movers_scanner.build_payload(crypto_movers, stock_movers)

        # Persist stats
        movers_scanner.persist_last_run(
            "crypto",
            {
                "count": len(crypto_movers),
                "ts": int(time.time()),
                "error": "",
                "duration_ms": 0
            },
            REDIS
        )
        movers_scanner.persist_last_run(
            "stocks",
            {
                "count": len(stock_movers),
                "ts": int(time.time()),
                "error": "",
                "duration_ms": 0
            },
            REDIS
        )

        return payload

    except Exception as e:
        LOGGER.error(f"scan_movers_error: {e}", exc_info=True)
        return JSONResponse(
            {"ok": False, "detail": str(e)[:200]},
            status_code=500
        )


@router.get("/api/scan/health")
async def api_scan_health():
    """
    Get movers scanner health status.

    Returns:
        {
            "last_crypto_ts": int,
            "last_stocks_ts": int,
            "last_counts": {...},
            "last_error": {...},
            "redis_dedup_stats": {...}
        }
    """
    try:
        from app.core import movers_scanner

        # Get last run stats
        crypto_stats = movers_scanner.get_last_run_stats("crypto", REDIS)
        stocks_stats = movers_scanner.get_last_run_stats("stocks", REDIS)

        # Get de-dup stats from Redis
        dedup_stats = {}
        if REDIS:
            try:
                # Count active de-dup keys
                date = datetime.now().strftime("%Y-%m-%d")
                pattern = f"ghost:alert:mover:*:{date}"

                cursor = 0
                dedup_count = 0
                while True:
                    cursor, keys = REDIS.scan(cursor, match=pattern, count=100)
                    dedup_count += len(keys)
                    if cursor == 0:
                        break

                dedup_stats = {
                    "active_dedups_today": dedup_count,
                    "pattern": pattern
                }
            except Exception as e:
                dedup_stats = {"error": str(e)}

        return {
            "last_crypto_ts": crypto_stats.get("ts") if crypto_stats else None,
            "last_stocks_ts": stocks_stats.get("ts") if stocks_stats else None,
            "last_counts": {
                "crypto": crypto_stats.get("count") if crypto_stats else 0,
                "stocks": stocks_stats.get("count") if stocks_stats else 0
            },
            "last_error": {
                "crypto": crypto_stats.get("error") if crypto_stats else "",
                "stocks": stocks_stats.get("error") if stocks_stats else ""
            },
            "redis_dedup_stats": dedup_stats,
            "ts": int(time.time() * 1000)
        }

    except Exception as e:
        LOGGER.error(f"scan_health_error: {e}", exc_info=True)
        return JSONResponse(
            {"ok": False, "detail": str(e)[:200]},
            status_code=500
        )


@router.get("/api/scan/stocks")
async def api_scan_stocks():
    """
    Scan entire stock market for opportunities.
    Returns top 20 high-confidence stock opportunities.
    """
    try:
        from core.market_scanner import scan_stocks

        opportunities = await scan_stocks()

        return {
            "ok": True,
            "opportunities": opportunities,
            "count": len(opportunities),
            "timestamp": int(time.time()),
        }
    except Exception as e:
        LOGGER.error(f"Stock scan failed: {e}", exc_info=True)
        return {
            "ok": False,
            "error": str(e),
            "opportunities": [],
            "count": 0,
            "timestamp": int(time.time()),
        }


@router.get("/api/scan/crypto")
async def api_scan_crypto():
    """
    Scan crypto market for opportunities.
    Returns high-confidence crypto opportunities.
    """
    try:
        from core.market_scanner import scan_crypto

        opportunities = await scan_crypto()

        return {
            "ok": True,
            "opportunities": opportunities,
            "count": len(opportunities),
            "timestamp": int(time.time()),
        }
    except Exception as e:
        LOGGER.error(f"Crypto scan failed: {e}", exc_info=True)
        return {
            "ok": False,
            "error": str(e),
            "opportunities": [],
            "count": 0,
            "timestamp": int(time.time()),
        }


@router.get("/api/scan/all")
async def api_scan_all():
    """
    Scan both stocks and crypto for opportunities.
    Returns combined opportunity list.
    """
    try:
        from core.market_scanner import scan_all

        results = await scan_all()

        return {"ok": True, **results}
    except Exception as e:
        LOGGER.error(f"Full market scan failed: {e}", exc_info=True)
        return {
            "ok": False,
            "error": str(e),
            "stocks": [],
            "crypto": [],
            "total": 0,
            "timestamp": int(time.time()),
        }


@router.get("/api/opportunities/top")
async def api_opportunities_top(limit: int = 10, min_confidence: float = None):
    """
    Get top-ranked opportunities across all markets with scoring.

    Query params:
        limit: Max opportunities to return (default 10)
        min_confidence: Minimum confidence threshold (default from MIN_ALERT_CONFIDENCE env)
    """
    # Use Railway env var if not specified
    if min_confidence is None:
        min_confidence = float(os.getenv("MIN_ALERT_CONFIDENCE", "0.55"))
    
    try:
        from core.market_scanner import scan_all
        from core.opportunity_scorer import rank_opportunities

        # Get all opportunities
        results = await scan_all()

        # Combine
        all_opportunities = results.get("stocks", []) + results.get("crypto", [])

        # Filter by confidence
        filtered = [
            opp for opp in all_opportunities if opp.get("confidence", 0) >= min_confidence
        ]

        # Calculate scores and rank
        ranked = rank_opportunities(filtered)

        # Take top N
        top = ranked[:limit]

        return {
            "ok": True,
            "opportunities": top,
            "count": len(top),
            "total_scanned": len(all_opportunities),
            "min_confidence": min_confidence,
            "timestamp": int(time.time()),
        }
    except Exception as e:
        LOGGER.error(f"Top opportunities failed: {e}", exc_info=True)
        return {
            "ok": False,
            "error": str(e),
            "opportunities": [],
            "count": 0,
            "timestamp": int(time.time()),
        }


@router.get("/api/opportunity/live")
async def api_opportunity_live():
    """
    Fast cached endpoint for live opportunities.
    Optimized for UI real-time updates (2s timeout max).
    """
    try:
        from core.market_scanner import scan_all
        from core.opportunity_scorer import rank_opportunities

        # Use cached scan if available (within 5min)
        results = await scan_all()
        all_opportunities = results.get("stocks", []) + results.get("crypto", [])

        # Quick rank (top 5 only for speed)
        ranked = rank_opportunities(all_opportunities)[:5]

        return {
            "ok": True,
            "opportunities": ranked,
            "count": len(ranked),
            "cached": True,
            "timestamp": int(time.time()),
        }
    except Exception as e:
        LOGGER.error(f"Live opportunity failed: {e}")
        return {
            "ok": False,
            "error": str(e),
            "opportunities": [],
            "count": 0,
            "timestamp": int(time.time()),
        }


@router.get("/opportunities")
async def opportunities_dashboard():
    """
    Serve Ghost Investment Hunter dashboard (opportunities UI).
    Shows high-confidence alerts, top movers, detected opportunities, and accuracy.
    """
    from fastapi.templating import Jinja2Templates

    templates = Jinja2Templates(directory="templates")

    # Create a mock request object with empty headers
    # FastAPI templates expect a Request object for rendering
    class MockRequest:
        def __init__(self):
            self.headers = {}
            self.path_params = {}

    try:
        return templates.TemplateResponse(
            "opportunities.html",
            {"request": MockRequest(), "active": "opportunities"}
        )
    except Exception as e:
        LOGGER.error(f"Opportunities dashboard failed: {e}")
        return HTMLResponse(
            content="""
            <html><head><title>Ghost Investment Hunter</title></head>
            <body><h1>Ghost Investment Hunter</h1>
            <p>Dashboard temporarily unavailable</p>
            <p><a href="/cockpit">Return to Cockpit</a></p></body></html>
            """,
            status_code=500
        )


