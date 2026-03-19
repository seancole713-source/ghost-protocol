"""Routes: cron — extracted from wolf_app.py (Step 12)"""
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

# --- 4 endpoints ---

@router.get("/cron/daily-scout")
@router.post("/cron/daily-scout")
async def cron_daily_scout(request: Request):
    """
    🌅 6:00 AM DAILY SCOUT
    
    External cron trigger for daily scouting.
    Scans all stocks + crypto and records trades.
    
    cron-job.org setup:
      URL: https://your-app.railway.app/cron/daily-scout
      Time: 12:00 UTC (6:00 AM Central)
      Method: GET
    """
    if not _validate_cron_request(request):
        LOGGER.warning("⚠️ [CRON] Invalid secret for daily-scout")
        return {"ok": False, "error": "Invalid cron secret"}
    
    try:
        from core.smart_scout import SmartScout
        
        LOGGER.info("🌅 [CRON] Running daily scout via external trigger...")
        scout = SmartScout()
        result = scout.full_scout()
        
        stocks = result.get("stocks", {}).get("scouted", 0)
        crypto = result.get("crypto", {}).get("scouted", 0)
        total = result.get("total_scouted", stocks + crypto)
        
        LOGGER.info(f"🌅 [CRON] Daily scout complete: {total} assets ({stocks} stocks, {crypto} crypto)")
        
        return {
            "ok": True,
            "job": "daily-scout",
            "timestamp": datetime.now(UTC).isoformat(),
            "stocks_scouted": stocks,
            "crypto_scouted": crypto,
            "total_scouted": total
        }
    except Exception as e:
        LOGGER.error(f"🌅 [CRON] Daily scout error: {e}")
        return {"ok": False, "error": str(e)}


@router.get("/cron/morning-alert")
@router.post("/cron/morning-alert")
async def cron_morning_alert(request: Request):
    """
    ☀️ 8:00 AM MORNING ALERT
    
    External cron trigger for TOP 10 Telegram alert.
    Sends the daily picks to Telegram.
    
    cron-job.org setup:
      URL: https://your-app.railway.app/cron/morning-alert
      Time: 14:00 UTC (8:00 AM Central)
      Method: GET
    """
    if not _validate_cron_request(request):
        LOGGER.warning("⚠️ [CRON] Invalid secret for morning-alert")
        return {"ok": False, "error": "Invalid cron secret"}
    
    try:
        from core.smart_scout import get_elite_predictions
        
        LOGGER.info("☀️ [CRON] Sending morning alert via external trigger...")
        
        elite = get_elite_predictions()
        stocks = elite.get("elite_stocks", [])[:5]
        crypto = elite.get("elite_crypto", [])[:5]
        
        if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
            msg = "☀️ <b>GHOST MORNING PICKS</b>\n\n"
            msg += f"📅 {datetime.now(UTC).strftime('%B %d, %Y')}\n\n"
            
            if stocks:
                msg += "📈 <b>Top 5 Stocks:</b>\n"
                for i, s in enumerate(stocks, 1):
                    msg += f"  {i}. {s}\n"
            else:
                msg += "📈 <i>No stock picks yet</i>\n"
            
            if crypto:
                msg += "\n🪙 <b>Top 5 Crypto:</b>\n"
                for i, c in enumerate(crypto, 1):
                    msg += f"  {i}. {c}\n"
            else:
                msg += "\n🪙 <i>No crypto picks yet</i>\n"
            
            msg += "\n<i>Triggered by external cron</i>"
            
            success = _tg_send_chat_message(TELEGRAM_CHAT_ID, msg)
            
            return {
                "ok": True,
                "job": "morning-alert",
                "timestamp": datetime.now(UTC).isoformat(),
                "telegram_sent": success,
                "stocks": stocks,
                "crypto": crypto
            }
        else:
            return {
                "ok": False,
                "error": "Telegram not configured",
                "stocks": stocks,
                "crypto": crypto
            }
            
    except Exception as e:
        LOGGER.error(f"☀️ [CRON] Morning alert error: {e}")
        return {"ok": False, "error": str(e)}


@router.get("/cron/evening-resolve")
@router.post("/cron/evening-resolve")
async def cron_evening_resolve(request: Request):
    """
    🌙 6:00 PM EVENING RESOLVE
    
    External cron trigger to resolve trades and update rankings.
    
    cron-job.org setup:
      URL: https://your-app.railway.app/cron/evening-resolve
      Time: 00:00 UTC (6:00 PM Central)
      Method: GET
    """
    if not _validate_cron_request(request):
        LOGGER.warning("⚠️ [CRON] Invalid secret for evening-resolve")
        return {"ok": False, "error": "Invalid cron secret"}
    
    try:
        LOGGER.info("🌙 [CRON] Resolving trades via external trigger...")
        
        # Import and run resolver
        try:
            from core.ghost_scout import GameResolver
            resolver = GameResolver()
            result = resolver.resolve_pending_trades(hours_old=24)
        except ImportError:
            from core.money_game_engine import get_money_game
            game = get_money_game()
            result = game.resolve_open_trades()
        
        LOGGER.info(f"🌙 [CRON] Trade resolution complete: {result}")
        
        return {
            "ok": True,
            "job": "evening-resolve",
            "timestamp": datetime.now(UTC).isoformat(),
            "result": result
        }
    except Exception as e:
        LOGGER.error(f"🌙 [CRON] Evening resolve error: {e}")
        return {"ok": False, "error": str(e)}


@router.get("/cron/health")
async def cron_health():
    """
    💓 CRON HEALTH CHECK
    
    Simple endpoint to verify the app is up.
    cron-job.org can ping this to ensure availability.
    """
    return {
        "ok": True,
        "timestamp": datetime.now(UTC).isoformat(),
        "message": "Ghost is alive and ready for cron jobs"
    }


