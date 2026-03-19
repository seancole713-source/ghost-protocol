"""Routes: alerts — extracted from wolf_app.py (Step 12)"""
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

# --- 32 endpoints ---

@router.post("/api/alerts/template")
async def api_alerts_template_post(
    body: AlertTemplateBody, credentials: HTTPAuthorizationCredentials | None = AUTH_DEP
):
    _require_bearer(
        (f"Bearer {credentials.credentials}") if credentials and credentials.credentials else None
    )
    if body.signal_template:
        ALERT_CONFIG["signal_template"] = body.signal_template.strip()
    if body.status_template:
        ALERT_CONFIG["status_template"] = body.status_template.strip()
    return {
        "ok": True,
        "signal_template": ALERT_CONFIG["signal_template"],
        "status_template": ALERT_CONFIG["status_template"],
    }


@router.get("/api/v3/alerts/status")
async def api_v3_alerts_status():
    """
    Get alert system status and recent alerts.
    
    Returns active alerts count, Telegram status, and recent notifications.
    """
    try:
        from wolf_app import STATE
        
        # Check if Telegram is configured
        telegram_configured = bool(os.getenv("TELEGRAM_BOT_TOKEN")) and bool(os.getenv("TELEGRAM_CHAT_ID"))
        
        # Get recent alert stats from state
        alert_count = STATE.get("alert_count", 0)
        last_alert_time = STATE.get("last_alert_time", 0)
        
        # Check if any predictions triggered alerts recently (last hour)
        recent_alerts = 0
        if last_alert_time > 0 and time.time() - last_alert_time < 3600:
            recent_alerts = alert_count
        
        return {
            "ok": True,
            "telegram_configured": telegram_configured,
            "telegram_enabled": telegram_configured,
            "alert_count_1h": recent_alerts,
            "last_alert_timestamp": last_alert_time if last_alert_time > 0 else None,
            "min_alert_confidence": 0.70,  # From touch_calibration_sqlite.py stage5/stage6 gates
            "min_confidence_threshold": float(os.getenv("MIN_ALERT_CONFIDENCE", "0.55")),  # Legacy field
            "instant_alert_threshold": int(os.getenv("INSTANT_ALERT_THRESHOLD", "80")),
            "status": "active" if telegram_configured else "not_configured",
            "timestamp": time.time()
        }
    
    except Exception as e:
        LOGGER.error(f"Alert status failed: {e}", exc_info=True)
        return {
            "ok": False,
            "error": str(e),
            "status": "error"
        }


@router.get("/api/v3/alerts/smart_cap")
async def api_v3_alerts_smart_cap():
    """
    Get Smart Cap status - daily alert limiting system.
    
    SMART CAP prevents spam by:
    - Limiting to 10 alerts per day (configurable)
    - Requiring 80%+ confidence minimum
    - Last 3 slots need 85%+ confidence
    - Last slot needs 90%+ confidence
    
    Returns current count, cap, and today's alerts log.
    """
    try:
        from core.telegram_alerts import (
            get_daily_alert_count, 
            get_daily_alert_log, 
            DAILY_ALERT_CAP,
            MIN_ALERT_CONFIDENCE,
            SMART_CAP_ENABLED
        )
        
        count = get_daily_alert_count()
        log = get_daily_alert_log()
        remaining = DAILY_ALERT_CAP - count
        
        # Determine current minimum required confidence
        if remaining <= 1:
            current_min = 0.90
        elif remaining <= 3:
            current_min = 0.85
        else:
            current_min = MIN_ALERT_CONFIDENCE
        
        return {
            "ok": True,
            "smart_cap_enabled": SMART_CAP_ENABLED,
            "daily_cap": DAILY_ALERT_CAP,
            "alerts_sent_today": count,
            "alerts_remaining": remaining,
            "min_confidence_base": MIN_ALERT_CONFIDENCE,
            "min_confidence_current": current_min,
            "alerts_log": log,
            "next_slot_requirement": (
                f"90%+ (last slot)" if remaining == 1 else
                f"85%+ (high conviction only)" if remaining <= 3 else
                f"{MIN_ALERT_CONFIDENCE:.0%}+ (standard)"
            ),
            "status": "capped" if remaining <= 0 else "accepting",
            "timestamp": time.time()
        }
    
    except Exception as e:
        LOGGER.error(f"Smart cap status failed: {e}", exc_info=True)
        return {
            "ok": False,
            "error": str(e)
        }


@router.post("/api/v3/alerts/test")
async def api_v3_alerts_test(channel: str = "slack", message: str = "Test from Ghost Protocol"):
    """
    Option 4: Test alert system - sends test message to specified channel.
    """
    try:
        from core.alert_system import send_trade_alert
        
        test_trade = {
            "symbol": "TEST",
            "side": "BUY",
            "quantity": 1,
            "price": 100.0,
            "pnl": 0.0,
            "note": message
        }
        
        await send_trade_alert(test_trade)
        
        return {
            "ok": True,
            "message": f"Test alert sent to {channel}",
            "timestamp": datetime.now(UTC).isoformat()
        }
    
    except Exception as e:
        return {"ok": False, "error": str(e)}


@router.post("/api/v3/alerts/test-all")
async def api_v3_alerts_test_all():
    """
    Option 4: Test all configured alert channels.
    """
    try:
        from core.alert_system import send_trade_alert, send_milestone_alert
        
        # Test trade alert
        await send_trade_alert({
            "symbol": "TEST",
            "side": "BUY",
            "quantity": 1,
            "price": 100.0,
            "pnl": 10.50
        })
        
        # Test milestone alert
        await send_milestone_alert("test", 100.0)
        
        return {
            "ok": True,
            "message": "Test alerts sent to all configured channels",
            "timestamp": datetime.now(UTC).isoformat()
        }
    
    except Exception as e:
        return {"ok": False, "error": str(e)}


@router.get("/api/telegram/reinit")
async def api_telegram_reinit():
    """Reinitialize Telegram connection (for auto-fixer)"""
    try:
        global TELEGRAM_BOT, TELEGRAM_CHAT_ID

        # Re-read credentials
        token = os.getenv("TELEGRAM_BOT_TOKEN")
        chat_id = os.getenv("TELEGRAM_CHAT_ID")

        if not token or not chat_id:
            return {"ok": False, "error": "Telegram credentials not configured"}

        # Reinitialize bot
        import telegram

        TELEGRAM_BOT = telegram.Bot(token=token)
        TELEGRAM_CHAT_ID = chat_id

        # Test connection
        try:
            bot_info = await TELEGRAM_BOT.get_me()
            return {
                "ok": True,
                "reinitialized": True,
                "bot_username": bot_info.username,
                "chat_id": chat_id,
            }
        except Exception as e:
            return {"ok": False, "error": f"Telegram test failed: {str(e)}"}

    except Exception as e:
        return {"ok": False, "error": str(e)}


@router.get("/alerts/selftest")
async def alerts_selftest():
    return {"ok": bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)}


@router.post("/api/telegram/test")
def api_telegram_test(payload: dict[str, Any] | None = None):
    """Build a signal card (with corporate action adjusted PnL) and optionally send.

    Request JSON (all optional):
      {
        "send": true|false,   # if true and credentials set, actually send
        "action": "BUY|SELL|HOLD",  # override action
        "price": 12.34,       # override price for preview
        "note": "extra line"  # append custom line
      }
    """
    try:
        t0 = time.perf_counter()
        send_flag = bool((payload or {}).get("send"))
        override_action = (payload or {}).get("action")
        override_price = (payload or {}).get("price")
        extra_note = (payload or {}).get("note")
        # Minimal synthetic signal dict
        sig = {
            "action": (override_action or "HOLD").upper(),
            "price": override_price,
            "provider": "test",
        }
        card = _signal_card(sig, include_trace=False)
        if extra_note:
            card += f"\n{extra_note}"
        sent = False
        if send_flag:
            sent = send_telegram(card)
        # Metrics
        try:
            if "_H_TG_TEST" in globals() and _H_TG_TEST is not None:
                _H_TG_TEST.observe(time.perf_counter() - t0)
            if "_C_TG_TEST" in globals() and _C_TG_TEST is not None:
                _C_TG_TEST.labels(sent=str(bool(sent)).lower()).inc()
        except Exception:
            pass
        return {
            "ok": True,
            "sent": bool(sent),
            "can_send": bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID),
            "card": card,
        }
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"telegram_test_error: {e}") from e


@router.get("/alerts/test")
@router.post("/alerts/test")
async def alerts_test(credentials: HTTPAuthorizationCredentials | None = AUTH_DEP, message: str = None, send: bool = True):
    # Skip auth for test endpoint (UI button should just work)
    if PROTECT_ALERTS_TEST:
        try:
            _require_bearer(
                (f"Bearer {credentials.credentials}")
                if credentials and credentials.credentials
                else None
            )
        except HTTPException:
            # Allow test to proceed even without auth for UI convenience
            pass

    # If alerts are not configured, report gracefully
    if not (TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID):
        return {"ok": False, "reason": "alerts-disabled"}

    # Send custom or default test message
    try:
        if message:
            test_msg = message
        else:
            test_msg = "🔔 Ghost Test Alert\n\n✅ UI → API → Telegram working!\n\nIf you see this, your alerts are configured correctly."
        
        if not send:
            return {"ok": True, "message": test_msg, "note": "Add ?send=true to actually send"}
        
        sent, deliveries = send_telegram_detailed(test_msg)
        if sent:
            return {"ok": True, "sent": True, "deliveries": deliveries}

        LOGGER.warning("alerts_test_send_failed", extra={"deliveries": deliveries})
        return {
            "ok": False,
            "sent": False,
            "error": "telegram_send_failed",
            "deliveries": deliveries,
        }
    except Exception as e:
        LOGGER.error(f"Test alert failed: {e}", exc_info=True)
        return {"ok": False, "sent": False, "error": str(e)}


@router.post("/alerts/predictions/send")
async def alerts_predictions_send():
    """
    🎯 Send current predictions as a formatted Telegram alert.
    
    Includes both stocks (trial mode) and crypto predictions.
    No authentication required for testing.
    """
    try:
        from datetime import datetime
        
        # Build prediction message from current state
        lines = ["🎯 <b>GHOST AI TRADING SIGNALS</b>"]
        lines.append(f"⏰ {datetime.now().strftime('%I:%M %p')} - Live Scan")
        lines.append("")
        
        # Get stock prediction (NVDA)
        try:
            from core.stock_engine import get_stock_engine
            engine = get_stock_engine()
            nvda = await engine.predict("NVDA", bypass_calendar=True)
            
            lines.append("<b>📈 STOCK PREDICTIONS</b>")
            lines.append("━━━━━━━━━━━━━━━━")
            
            direction_emoji = "⬆️" if nvda.direction == "UP" else ("⬇️" if nvda.direction == "DOWN" else "➡️")
            lines.append(f"<b>NVDA</b> - NVIDIA {direction_emoji}")
            lines.append(f"   Confidence: {nvda.confidence*100:.0f}%")
            lines.append(f"   Entry: ${nvda.entry_price:.2f}")
            lines.append(f"   Target: ${nvda.target_price:.2f}")
            if nvda.reasons:
                intel_reasons = [r for r in nvda.reasons if "Intel:" in r or "Ensemble" in r]
                if intel_reasons:
                    lines.append(f"   Intel: {', '.join(r.replace('Intel: ', '') for r in intel_reasons[:2])}")
            lines.append("")
        except Exception as e:
            lines.append(f"<b>📈 STOCKS:</b> Error - {str(e)[:50]}")
            lines.append("")
        
        # Get crypto predictions from cache
        lines.append("<b>💎 CRYPTO PREDICTIONS</b>")
        lines.append("━━━━━━━━━━━━━━━━")
        
        crypto_count = 0
        for sym, pred in list(_LATEST_PREDICTIONS.items())[:5]:
            if pred.get("direction") in ("UP", "DOWN") and pred.get("confidence", 0) >= 0.70:
                direction = pred.get("direction")
                direction_emoji = "⬆️" if direction == "UP" else "⬇️"
                conf = pred.get("confidence", 0) * 100
                
                # Try multiple field names for entry price
                entry = pred.get("entry_price") or pred.get("price_at_prediction") or pred.get("price") or 0
                target = pred.get("target_price") or pred.get("take_profit") or 0
                
                lines.append(f"<b>{sym}</b> {direction_emoji}")
                lines.append(f"   Confidence: {conf:.0f}%")
                if entry > 0:
                    lines.append(f"   Entry: ${entry:.2f}" if entry > 1 else f"   Entry: ${entry:.4f}")
                    if target > 0:
                        pct_change = ((target - entry) / entry) * 100
                        lines.append(f"   Target: ${target:.2f} ({pct_change:+.1f}%)" if target > 1 else f"   Target: ${target:.4f} ({pct_change:+.1f}%)")
                lines.append("")
                crypto_count += 1
        
        if crypto_count == 0:
            lines.append("   No high-confidence signals (>70%) right now")
            lines.append("")
        
        # Footer
        lines.append("━━━━━━━━━━━━━━━━")
        lines.append("📊 <i>Ghost AI - V2 Quality Filtered</i>")
        lines.append("🔧 <i>Stock Engine v1 + Intel + Ensemble</i>")
        
        message = "\n".join(lines)
        
        # Send via Telegram
        sent, deliveries = send_telegram_detailed(message)
        
        return {
            "ok": sent,
            "message_preview": message[:500],
            "deliveries": deliveries
        }
        
    except Exception as e:
        LOGGER.error(f"Prediction alert failed: {e}", exc_info=True)
        return {"ok": False, "error": str(e)}


@router.get("/alerts/top10/status")
async def top10_status():
    """Get TOP 10 aggregator status - how many picks are queued"""
    try:
        from core.top10_aggregator import get_top10_aggregator
        aggregator = get_top10_aggregator()
        status = aggregator.get_status()
        return {
            "ok": True,
            "top10_aggregator_enabled": os.getenv("TOP10_AGGREGATOR_ENABLED", "1") == "1",
            "individual_alerts_enabled": os.getenv("INDIVIDUAL_ALERTS_ENABLED", "0") == "1",
            **status
        }
    except Exception as e:
        LOGGER.error(f"TOP 10 status error: {e}")
        return {"ok": False, "error": str(e)}


@router.post("/alerts/backfill-outcomes")
async def backfill_no_data_outcomes(
    batch_size: int = 100,
    authorization: Optional[str] = Header(None),
    x_cron_secret: Optional[str] = Header(None, alias="X-Cron-Secret"),
):
    """
    BACKFILL: Re-evaluate 'no_data' outcomes using price from features_json.
    
    The reconciler was failing 97.8% of the time because it couldn't fetch
    historical prices. But the predictions table HAS the entry price stored
    in features_json.current_price!
    
    This endpoint:
    1. Finds outcomes with status='no_data' 
    2. Looks up the original prediction to get features_json.current_price
    3. Fetches the current price for resolution
    4. Calculates actual_direction and hit_direction
    5. Updates the outcome to status='completed'
    """
    import psycopg2
    import json
    
    # Auth check
    valid_auth = False
    if authorization and authorization.startswith("Bearer "):
        token = authorization.replace("Bearer ", "")
        if token == os.getenv("GHOST_SECRET_TOKEN", "ghost-prod-2024"):
            valid_auth = True
    if x_cron_secret and x_cron_secret == os.getenv("CRON_SECRET", "ghost-cron-2024"):
        valid_auth = True
    if not valid_auth:
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        return {"ok": False, "error": "DATABASE_URL not configured"}
    
    try:
        from core.db_pool import get_sync_connection
        with get_sync_connection() as conn:
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            # Get 'no_data' outcomes that have corresponding predictions
            cur.execute("""
                SELECT 
                    o.prediction_id,
                    o.symbol,
                    o.predicted_direction,
                    o.closed_at,
                    p.run_at,
                    p.features_json,
                    p.confidence
                FROM ghost_prediction_outcomes o
                JOIN predictions p ON o.prediction_id = p.id
                WHERE o.status = 'no_data'
                LIMIT %s
            """, (batch_size,))
            
            no_data_outcomes = cur.fetchall()
            
            if not no_data_outcomes:
                return {
                    "ok": True,
                    "message": "No 'no_data' outcomes to backfill",
                    "processed": 0
                }
            
            processed = 0
            skipped = 0
            errors = []
            
            # Direction threshold
            DIRECTION_THRESHOLD_PCT = float(os.getenv("ACCURACY_DIRECTION_THRESHOLD_PCT", "0.25"))
            
            for outcome in no_data_outcomes:
                try:
                    # Extract price_at_prediction from features_json
                    features_json = outcome.get("features_json")
                    if not features_json:
                        skipped += 1
                        continue
                    
                    features = json.loads(features_json) if isinstance(features_json, str) else features_json
                    price_t0 = features.get("current_price") or features.get("PRICE")
                    
                    if not price_t0:
                        skipped += 1
                        continue
                    
                    price_t0 = float(price_t0)
                    symbol = outcome["symbol"]
                    pred_id = outcome["prediction_id"]
                    pred_direction = outcome["predicted_direction"]
                    pred_confidence = outcome.get("confidence", 0.5)
                    
                    # Get current price for resolution
                    from services.outcome_reconciler_v2 import get_symbol_price
                    price_t1 = get_symbol_price(symbol)
                    
                    if price_t1 is None:
                        skipped += 1
                        continue
                    
                    # Compute realized movement
                    realized_move_pct = ((price_t1 - price_t0) / price_t0) * 100
                    
                    # Determine actual direction
                    if realized_move_pct > DIRECTION_THRESHOLD_PCT:
                        actual_direction = "UP"
                    elif realized_move_pct < -DIRECTION_THRESHOLD_PCT:
                        actual_direction = "DOWN"
                    else:
                        actual_direction = "FLAT"
                    
                    # Determine if prediction was correct
                    hit_direction = 1 if actual_direction == pred_direction else 0
                    
                    # Update the outcome
                    cur.execute("""
                        UPDATE ghost_prediction_outcomes
                        SET 
                            price_at_prediction = %s,
                            price_at_resolution = %s,
                            realized_move_pct = %s,
                            actual_direction = %s,
                            hit_direction = %s,
                            status = 'completed',
                            notes = 'Backfilled from features_json'
                        WHERE prediction_id = %s
                    """, (
                        price_t0,
                        price_t1,
                        realized_move_pct,
                        actual_direction,
                        hit_direction,
                        pred_id
                    ))
                    
                    processed += 1
                    
                except Exception as e:
                    errors.append(f"pred {outcome['prediction_id']}: {str(e)[:50]}")
            
            cur.close()
            
            return {
                "ok": True,
                "message": f"Backfilled {processed} outcomes",
                "processed": processed,
                "skipped": skipped,
                "errors": errors[:10] if errors else []
            }
        
    except Exception as e:
        import traceback
        return {"ok": False, "error": str(e), "traceback": traceback.format_exc()}


@router.post("/alerts/reconcile/now")
async def reconcile_predictions_now(request: Request):
    """
    🎯 ON-DEMAND PREDICTION RECONCILIATION
    
    Called by cron-job.org to process predictions that are now 48+ hours old.
    SECURED: Requires X-Cron-Secret header matching CRON_SECRET env var.
    
    Process:
    1. Call the existing outcome_reconciler_v2 to process pending predictions
    2. Compute per-symbol accuracy from the outcomes table
    3. Update ghost_symbol_accuracy table for TOP 10 learning
    
    Returns:
        - reconciler_result: Result from the outcome reconciler
        - symbol_accuracy_updated: How many symbols had their accuracy computed
    """
    from datetime import datetime
    
    # Check cron secret for authentication
    cron_secret = os.getenv("CRON_SECRET", "ghost-cron-2024")
    provided_secret = request.headers.get("X-Cron-Secret", "")
    
    # Allow bypass for local testing or if already authenticated via bearer token
    bypass_auth = os.getenv("BYPASS_CRON_AUTH", "false") == "true"
    
    # Also allow if request has valid bearer token (already authenticated)
    auth_header = request.headers.get("Authorization", "")
    has_valid_bearer = auth_header.startswith("Bearer ") and auth_header[7:] == os.getenv("API_SECRET", "ghost-prod-2024")
    
    if not bypass_auth and not has_valid_bearer and cron_secret and provided_secret != cron_secret:
        LOGGER.warning(f"[RECONCILE] Unauthorized reconcile attempt")
        return {"ok": False, "error": "Unauthorized - invalid X-Cron-Secret"}
    
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        return {"ok": False, "error": "DATABASE_URL not configured"}
    
    try:
        # Step 1: Run the existing outcome reconciler
        reconciler_result = {"skipped": True, "reason": "reconciler not run"}
        try:
            from services.outcome_reconciler_v2 import reconcile_outcomes_v2
            reconciler_result = reconcile_outcomes_v2()
            LOGGER.info(f"[RECONCILE] Outcome reconciler result: {reconciler_result}")
        except Exception as rec_err:
            LOGGER.error(f"[RECONCILE] Reconciler error: {rec_err}")
            reconciler_result = {"error": str(rec_err)}
        
        # Step 2: Compute symbol accuracy from outcomes table
        from core.db_pool import get_sync_connection
        with get_sync_connection() as conn:
            cur = conn.cursor()
            
            # Create ghost_symbol_accuracy table if it doesn't exist
            cur.execute("""
                CREATE TABLE IF NOT EXISTS ghost_symbol_accuracy (
                    symbol VARCHAR(20) PRIMARY KEY,
                    total_predictions INTEGER DEFAULT 0,
                    correct_predictions INTEGER DEFAULT 0,
                    accuracy_pct NUMERIC(5, 2) DEFAULT 0,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Check if we're in INVERSE_GHOST mode - FIXED: Use INVERSE_GHOST (not INVERSE_GHOST_MODE) - default to OFF (0)
            inverse_mode = os.getenv("INVERSE_GHOST", "0") == "1"
            
            # Compute per-symbol accuracy from ghost_prediction_outcomes table
            # In INVERSE_GHOST mode, hit_direction=0 (raw wrong) is actually CORRECT
            # because we invert the raw predictions
            if inverse_mode:
                # Inverted accuracy: count when hit_direction=0 as correct
                cur.execute("""
                    SELECT 
                        symbol,
                        COUNT(*) as total,
                        SUM(CASE WHEN hit_direction = 0 THEN 1 ELSE 0 END) as correct
                    FROM ghost_prediction_outcomes
                    WHERE symbol IS NOT NULL
                    GROUP BY symbol
                    HAVING COUNT(*) >= 1
                """)
            else:
                # Normal mode: hit_direction=1 means correct
                cur.execute("""
                    SELECT 
                        symbol,
                        COUNT(*) as total,
                        SUM(CASE WHEN hit_direction = 1 THEN 1 ELSE 0 END) as correct
                    FROM ghost_prediction_outcomes
                    WHERE symbol IS NOT NULL
                    GROUP BY symbol
                    HAVING COUNT(*) >= 1
                """)
            
            symbol_stats = cur.fetchall()
            symbols_updated = 0
            
            for stat in symbol_stats:
                symbol, total, correct = stat
                accuracy = (correct / total * 100) if total > 0 else 0
                
                cur.execute("""
                    INSERT INTO ghost_symbol_accuracy (symbol, total_predictions, correct_predictions, accuracy_pct, last_updated)
                    VALUES (%s, %s, %s, %s, NOW())
                    ON CONFLICT (symbol) DO UPDATE SET
                        total_predictions = EXCLUDED.total_predictions,
                        correct_predictions = EXCLUDED.correct_predictions,
                        accuracy_pct = EXCLUDED.accuracy_pct,
                        last_updated = NOW()
                """, (symbol, total, correct, accuracy))
                symbols_updated += 1
            
            # Get summary stats
            cur.execute("""
                SELECT 
                    COUNT(*) as symbols_tracked,
                    AVG(accuracy_pct) as avg_accuracy,
                    COUNT(CASE WHEN accuracy_pct < 40 AND total_predictions >= 10 THEN 1 END) as excluded_count,
                    COUNT(CASE WHEN accuracy_pct >= 70 AND total_predictions >= 10 THEN 1 END) as boosted_count
                FROM ghost_symbol_accuracy
            """)
            summary = cur.fetchone()
            
            # Get lists of excluded and boosted symbols
            cur.execute("""
                SELECT symbol, accuracy_pct, total_predictions
                FROM ghost_symbol_accuracy
                WHERE total_predictions >= 10 AND accuracy_pct < 40
                ORDER BY accuracy_pct ASC
                LIMIT 20
            """)
            excluded = [{"symbol": r[0], "accuracy": float(r[1]), "predictions": r[2]} for r in cur.fetchall()]
            
            cur.execute("""
                SELECT symbol, accuracy_pct, total_predictions
                FROM ghost_symbol_accuracy
                WHERE total_predictions >= 10 AND accuracy_pct >= 70
                ORDER BY accuracy_pct DESC
                LIMIT 20
            """)
            boosted = [{"symbol": r[0], "accuracy": float(r[1]), "predictions": r[2]} for r in cur.fetchall()]
        
        LOGGER.info(f"[RECONCILE] ✅ Updated accuracy for {symbols_updated} symbols")
        
        return {
            "ok": True,
            "reconciler_result": reconciler_result,
            "symbol_accuracy_updated": symbols_updated,
            "summary": {
                "symbols_tracked": summary[0] if summary else 0,
                "avg_accuracy_pct": float(summary[1]) if summary and summary[1] else 0,
                "excluded_from_top10": summary[2] if summary else 0,
                "boosted_in_top10": summary[3] if summary else 0,
            },
            "excluded_symbols": excluded,
            "boosted_symbols": boosted,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        import traceback
        LOGGER.error(f"[RECONCILE] Error: {e}")
        return {"ok": False, "error": str(e), "traceback": traceback.format_exc()}


@router.post("/alerts/top10/send")
async def top10_force_send():
    """Force send the TOP 10 message with whatever picks are queued"""
    try:
        from core.top10_aggregator import get_top10_aggregator
        aggregator = get_top10_aggregator()
        
        # Set telegram function
        def _send_telegram(msg: str) -> bool:
            return _tg_send_chat_message(TELEGRAM_CHAT_ID, msg)
        
        aggregator.set_telegram_func(_send_telegram)
        
        success = aggregator.force_send()
        return {
            "ok": success,
            "message": "TOP 10 sent" if success else "No picks queued"
        }
    except Exception as e:
        LOGGER.error(f"TOP 10 force send error: {e}")
        return {"ok": False, "error": str(e)}


@router.post("/alerts/top10/now")
async def top10_send_now(request: Request):
    """
    DISABLED (Feb 21, 2026): This cron endpoint was sending a SECOND TOP 10 card
    using hardcoded symbols (NVDA, META, PLTR...) and a legacy 0.50 confidence floor,
    completely bypassing the main notification loop's clean V3 pipeline + 0.70 floor.
    Result: duplicate cards every morning at 8:00 + 8:02 with different picks.
    
    The main notification loop in _post_startup_init() handles 8 AM sends correctly
    with proper edge whitelist, V3Filter, PostgreSQL dedup, and 0.70 floor.
    
    Kill the cron-job.org job too — this endpoint now returns immediately.
    """
    return {
        "ok": False,
        "disabled": True,
        "reason": "Cron TOP 10 disabled Feb 21 2026 — main notification loop handles 8 AM sends. Kill the cron-job.org job.",
    }

    # --- DEAD CODE BELOW (kept for archaeology) ---
    # Check cron secret for authentication
    cron_secret = os.getenv("CRON_SECRET", "ghost-cron-2024")
    provided_secret = request.headers.get("X-Cron-Secret", "")
    
    if not cron_secret:
        LOGGER.warning("[TOP10] CRON_SECRET not configured - endpoint disabled")
        return {"ok": False, "error": "CRON_SECRET not configured"}
    
    if provided_secret != cron_secret:
        LOGGER.warning(f"[TOP10] Invalid cron secret attempt from {request.client.host if request.client else 'unknown'}")
        return {"ok": False, "error": "Unauthorized - invalid X-Cron-Secret"}
    
    try:
        from core.ghost_notifications import get_notification_system, get_central_time
        import sqlite3
        
        LOGGER.info("[TOP10] Authenticated cron request - checking daily limit...")
        
        # Check if already sent today using database
        today = get_central_time().strftime("%Y-%m-%d")
        db_path = os.getenv("GHOST_TRACKING_DB", "data/ghost_tracking.db")
        
        try:
            conn = sqlite3.connect(db_path)
            already_sent = conn.execute(
                "SELECT COUNT(*) FROM notification_log WHERE notification_type = 'top10' AND DATE(sent_at) = ?",
                (today,)
            ).fetchone()[0]
            conn.close()
            
            if already_sent > 0:
                LOGGER.info(f"[TOP10] ⛔ Already sent today ({today}) - skipping duplicate")
                return {
                    "ok": False,
                    "message": f"TOP 10 already sent today ({today}). Only ONE per day.",
                    "already_sent_today": True,
                    "predictions_available": len(_LATEST_PREDICTIONS),
                }
        except Exception as db_err:
            LOGGER.warning(f"[TOP10] DB check failed: {db_err} - proceeding without dedup check")
        
        LOGGER.info("[TOP10] ✅ Not sent today - REDIRECTING TO CLEAN ENDPOINT...")
        
        # Jan 30, 2026: Use the CLEAN hardcoded symbols endpoint instead of _LATEST_PREDICTIONS
        # This prevents wrong symbols from Money Game from appearing
        try:
            clean_result = await send_top10_now_endpoint()  # Our clean /debug/send-top10-now
            success = clean_result.get("telegram_sent", False)
        except Exception as clean_err:
            LOGGER.error(f"[TOP10] Clean endpoint failed: {clean_err}")
            success = False
        
        # Log to database
        if success:
            try:
                conn = sqlite3.connect(db_path)
                conn.execute(
                    "INSERT INTO notification_log (notification_type, message_preview) VALUES (?, ?)",
                    ("top10", f"Sent {len(_LATEST_PREDICTIONS)} predictions")
                )
                conn.commit()
                conn.close()
                LOGGER.info(f"[TOP10] ✅ Logged to DB - won't send again today")
            except Exception as log_err:
                LOGGER.warning(f"[TOP10] Failed to log to DB: {log_err}")
        
        return {
            "ok": success,
            "message": "TOP 10 sent!" if success else "Failed to send or no predictions",
            "predictions_available": len(_LATEST_PREDICTIONS),
        }
        
    except Exception as e:
        LOGGER.error(f"TOP 10 NOW error: {e}", exc_info=True)
        return {"ok": False, "error": str(e)}


@router.post("/alerts/top10/force")
async def top10_force_send(request: Request):
    """
    🔧 FORCE send TOP 10 message - BYPASSES daily limit.
    
    Use this for testing only. Requires X-Cron-Secret AND X-Force-Send: true headers.
    """
    cron_secret = os.getenv("CRON_SECRET", "ghost-cron-2024")
    provided_secret = request.headers.get("X-Cron-Secret", "")
    force_header = request.headers.get("X-Force-Send", "")
    
    if provided_secret != cron_secret or force_header.lower() != "true":
        return {"ok": False, "error": "Requires X-Cron-Secret AND X-Force-Send: true"}
    
    try:
        LOGGER.warning("[TOP10] ⚠️ FORCE SEND - bypassing daily limit!")
        
        # Jan 30, 2026: Use CLEAN hardcoded symbols endpoint
        clean_result = await send_top10_now_endpoint()
        success = clean_result.get("telegram_sent", False)
        
        return {
            "ok": success,
            "message": "FORCE sent TOP 10!" if success else "Failed",
            "warning": "Daily limit bypassed - use sparingly",
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


@router.post("/alerts/watchdog/check")
async def watchdog_check_updates(request: Request):
    """
    🐺 WATCHDOG: Check tracked picks for significant moves (>3%) or target/stop hits.
    
    SECURED: Requires X-Cron-Secret header matching CRON_SECRET env var.
    Schedule this via cron-job.org at 12 PM, 4 PM, 8 PM Central.
    
    What it does:
    - Checks all picks from morning's TOP 10
    - Sends ALERT if target or stop is hit
    - Sends UPDATE if any pick moved >3%
    - At 8 AM: Also sends TOP 10 if not sent today (auto-integration)
    
    ⚡ PERFORMANCE: Returns 200 OK immediately, processes in background to avoid cron timeout.
    """
    # Check cron secret for authentication
    cron_secret = os.getenv("CRON_SECRET", "ghost-cron-2024")
    provided_secret = request.headers.get("X-Cron-Secret", "")
    
    if not cron_secret:
        LOGGER.warning("[WATCHDOG] CRON_SECRET not configured - endpoint disabled")
        return {"ok": False, "error": "CRON_SECRET not configured"}
    
    if provided_secret != cron_secret:
        LOGGER.warning(f"[WATCHDOG] Invalid cron secret attempt from {request.client.host if request.client else 'unknown'}")
        return {"ok": False, "error": "Unauthorized - invalid X-Cron-Secret"}
    
    # ⚡ CRITICAL: Return 200 OK immediately to avoid cron timeout
    # The actual check runs in background via asyncio.create_task()
    LOGGER.info("[WATCHDOG] 🐺 Authenticated cron request - scheduling background check...")
    
    # Schedule background task
    asyncio.create_task(_watchdog_background_check())
    
    # Return immediately (prevents cron-job.org 30s timeout)
    return {
        "ok": True,
        "message": "Watchdog check scheduled in background",
        "note": "Check logs for results - this endpoint returns immediately to avoid timeout"
    }


@router.get("/alerts/notifications/status")
async def notifications_status():
    """Get status of the new Ghost Notification System"""
    try:
        from core.ghost_notifications import get_notification_system, get_central_time
        
        notif = get_notification_system()
        
        # Auto-retry PostgreSQL if using SQLite (it might be available now)
        if not notif._use_postgres:
            notif.retry_postgres_connection()
        
        status = notif.get_status()
        
        return {
            "ok": True,
            "system": "ghost_notifications",
            **status,
            "predictions_in_memory": len(_LATEST_PREDICTIONS),
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


@router.post("/alerts/notifications/retry-postgres")
async def notifications_retry_postgres():
    """Retry PostgreSQL connection if currently using SQLite fallback"""
    try:
        from core.ghost_notifications import get_notification_system
        
        notif = get_notification_system()
        was_postgres = notif._use_postgres
        
        if was_postgres:
            return {
                "ok": True,
                "message": "Already using PostgreSQL",
                "database": "postgres",
                "persistent": True
            }
        
        # Try to get DATABASE_URL to diagnose
        db_url = os.getenv("DATABASE_URL", "")
        db_url_exists = bool(db_url)
        db_url_len = len(db_url) if db_url else 0
        
        success = notif.retry_postgres_connection()
        
        return {
            "ok": success,
            "message": "Switched to PostgreSQL" if success else "PostgreSQL connection failed - still using SQLite",
            "database": "postgres" if success else "sqlite",
            "persistent": success,
            "database_url_exists": db_url_exists,
            "database_url_length": db_url_len,
            "last_postgres_error": getattr(notif, '_last_postgres_error', None),
        }
    except Exception as e:
        import traceback
        return {"ok": False, "error": str(e), "traceback": traceback.format_exc()}


@router.get("/alerts/notifications/debug")
async def notifications_debug():
    """DEBUG: See exactly what the notification system sees"""
    try:
        from core.ghost_notifications import get_notification_system
        from core.asset_classifier import get_asset_type
        
        notif = get_notification_system()
        stocks, crypto = notif.get_top10_predictions(_LATEST_PREDICTIONS)
        
        # Also check raw classification
        raw_crypto = []
        for symbol, pred in list(_LATEST_PREDICTIONS.items())[:50]:
            if isinstance(pred, dict):
                asset = get_asset_type(symbol)
                if asset.startswith('crypto'):
                    raw_crypto.append({
                        'symbol': symbol,
                        'asset_type': asset,
                        'confidence': pred.get('confidence', 0),
                        'price': pred.get('price') or pred.get('current_price') or pred.get('entry_price'),
                    })
        
        return {
            "ok": True,
            "stocks_found": len(stocks),
            "crypto_found": len(crypto),
            "stocks": [{"symbol": s["symbol"], "conf": s["confidence"]} for s in stocks],
            "crypto": [{"symbol": c["symbol"], "conf": c["confidence"]} for c in crypto],
            "raw_crypto_in_predictions": raw_crypto[:10],
        }
    except Exception as e:
        import traceback
        return {"ok": False, "error": str(e), "traceback": traceback.format_exc()}


@router.post("/alerts/top10/reset")
async def top10_reset():
    """Reset the TOP 10 aggregator (clear queue, allow new TOP 10 today)"""
    try:
        from core.top10_aggregator import get_top10_aggregator
        aggregator = get_top10_aggregator()
        aggregator.reset()
        return {"ok": True, "message": "TOP 10 aggregator reset"}
    except Exception as e:
        LOGGER.error(f"TOP 10 reset error: {e}")
        return {"ok": False, "error": str(e)}


@router.get("/alerts/learning/test-exclusion")
async def test_learning_exclusion(symbol: str = "CHZ"):
    """Debug: Test if a symbol would be excluded by the learning filter"""
    try:
        from core.ghost_notifications import (
            should_exclude_symbol, 
            get_symbol_accuracy_from_postgres,
            HARDCODED_EXCLUSIONS,
            _ENV_EXCLUSIONS
        )
        from core.v2_quality import get_quality_system
        
        # Get accuracy data
        accuracy_data = get_symbol_accuracy_from_postgres()
        
        # Check should_exclude
        should_exclude, reason = should_exclude_symbol(symbol.upper(), accuracy_data)
        
        # Check V2
        v2 = get_quality_system()
        v2_should_predict, v2_reason = v2.should_predict(symbol.upper(), 0.85)
        
        return {
            "ok": True,
            "symbol": symbol.upper(),
            "learning_exclude": should_exclude,
            "learning_reason": reason,
            "in_hardcoded_exclusions": symbol.upper() in HARDCODED_EXCLUSIONS,
            "in_env_exclusions": symbol.upper() in _ENV_EXCLUSIONS,
            "v2_should_predict": v2_should_predict,
            "v2_reason": v2_reason,
            "v2_whitelisted": symbol.upper() in v2._whitelist,
            "v2_blacklisted": symbol.upper() in v2._blacklist,
            "accuracy_data": accuracy_data.get(symbol.upper(), "no_data"),
        }
    except Exception as e:
        import traceback
        return {"ok": False, "error": str(e), "traceback": traceback.format_exc()}


@router.get("/alerts/top10/debug")
async def top10_debug():
    """Debug endpoint - see what predictions are available for TOP 10"""
    try:
        from core.asset_classifier import get_asset_type
        
        min_conf = float(os.getenv("GHOST_TOP_10_MIN_CONF", "0.85"))
        
        all_preds = []
        for symbol, pred in list(_LATEST_PREDICTIONS.items()):
            if not isinstance(pred, dict):
                continue
            
            confidence = pred.get("confidence", 0)
            asset_class = get_asset_type(symbol)
            
            all_preds.append({
                "symbol": symbol,
                "confidence": confidence,
                "direction": pred.get("direction"),
                "asset_type": asset_class,
                "passes_min_conf": confidence >= min_conf,
                "price": pred.get("price") or pred.get("entry_price") or pred.get("current_price"),
            })
        
        # Sort by confidence
        all_preds.sort(key=lambda x: x["confidence"], reverse=True)
        
        high_conf = [p for p in all_preds if p["passes_min_conf"]]
        
        return {
            "ok": True,
            "total_predictions": len(all_preds),
            "high_confidence_count": len(high_conf),
            "min_confidence_threshold": min_conf,
            "top_20": all_preds[:20],
            "high_conf_picks": high_conf[:10],
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


@router.post("/telegram/webhook")
async def telegram_webhook(update: TelegramUpdate):
    """Receive Telegram updates and reply with status/signal on simple commands.
    Also handles natural language questions using Ghost AI.

    To set webhook (example):
      https://api.telegram.org/bot<token>/setWebhook?url=https://your.host/telegram/webhook
    """
    if not TELEGRAM_BOT_TOKEN:
        raise HTTPException(400, "telegram disabled")
    msg = update.message or {}
    chat = msg.get("chat") or {}
    chat_id = str(chat.get("id") or "")
    text = str(msg.get("text") or "").strip()
    if not chat_id:
        return {"ok": True}

    # Handle commands and natural questions
    try:
        if text.lower().startswith("/status"):
            # Build compact status
            price, prev, provider = get_wolf_price()
            q, a = _get_portfolio_qty_and_avg()  # Use helper to get correct values
            mv = q * (price if price is not None else a)
            reply = (
                "📊 WOLF Status\n"
                f"Qty: {q:.4f}\nAvg: ${a:.2f}\nPrice: {('?' if price is None else f'${price:.2f}')} ({provider or 'unavailable'})\n"
                f"NAV: ${mv:.2f}"
            )
            _tg_send_chat_message(chat_id, reply)
        elif text.lower().startswith("/signal"):
            sig = _evaluate_signal()
            card = _signal_card(sig, include_trace=False)
            card += (
                "\n\n🕒 SCHEDULED PREDICTIONS:\n"
                "  /todaypred - Send today's pre-market snapshot now\n\n"
            )
            _tg_send_chat_message(chat_id, card)
        elif text.lower().startswith("/pnl") or text.lower().startswith("/today"):
            # Calculate daily P&L
            price, prev, provider = get_wolf_price()
            q, _ = _get_portfolio_qty_and_avg()  # Use helper to get correct quantity

            if price is not None and prev is not None and q > 0:
                current_nav = q * price
                prev_nav = q * prev
                pnl = current_nav - prev_nav
                pnl_pct = (pnl / prev_nav * 100) if prev_nav > 0 else 0.0

                status_emoji = "📈" if pnl >= 0 else "📉"
                result = "WON" if pnl >= 0 else "LOST"
                sign = "+" if pnl >= 0 else ""

                reply = (
                    f"{status_emoji} Daily P&L\n"
                    f"Result: {result}\n"
                    f"Change: {sign}${pnl:.2f} ({sign}{pnl_pct:.2f}%)\n"
                    f"Previous: ${prev_nav:.2f}\n"
                    f"Current: ${current_nav:.2f}\n"
                    f"Price: ${prev:.2f} → ${price:.2f}"
                )
            elif q == 0:
                reply = "📊 No position held"
            else:
                reply = "⚠️ Price data unavailable for P&L calculation"

            _tg_send_chat_message(chat_id, reply)
        elif text.lower().startswith("/positions"):
            # Show all open positions
            try:
                from core.alpaca_broker import get_broker

                broker = get_broker()

                if not broker.enabled:
                    _tg_send_chat_message(chat_id, "⚠️ Broker not enabled")
                    return {"ok": True}

                positions = broker.get_positions()

                if not positions:
                    _tg_send_chat_message(chat_id, "📊 No open positions")
                else:
                    reply_lines = ["📊 Open Positions:\n"]
                    total_value = 0
                    total_pl = 0

                    for pos in positions:
                        symbol = pos.get("symbol")
                        qty = float(pos.get("qty", 0))
                        entry = float(pos.get("avg_entry_price", 0))
                        current = float(pos.get("current_price", 0))
                        pl = float(pos.get("unrealized_pl", 0))
                        pl_pct = float(pos.get("unrealized_plpc", 0)) * 100
                        value = float(pos.get("market_value", 0))

                        total_value += value
                        total_pl += pl

                        pl_emoji = "📈" if pl >= 0 else "📉"
                        sign = "+" if pl >= 0 else ""

                        reply_lines.append(
                            f"{pl_emoji} {symbol}: {qty:.2f} @ ${entry:.2f}\n"
                            f"   Current: ${current:.2f} ({sign}{pl_pct:.2f}%)\n"
                            f"   P&L: {sign}${pl:.2f}\n"
                        )

                    total_pl_pct = (
                        (total_pl / (total_value - total_pl) * 100)
                        if (total_value - total_pl) > 0
                        else 0
                    )
                    sign = "+" if total_pl >= 0 else ""
                    reply_lines.append(
                        f"\n💰 Total Value: ${total_value:.2f}\n"
                        f"💵 Total P&L: {sign}${total_pl:.2f} ({sign}{total_pl_pct:.2f}%)"
                    )

                    _tg_send_chat_message(chat_id, "".join(reply_lines))
            except Exception as e:
                _tg_send_chat_message(chat_id, f"❌ Error: {str(e)[:100]}")
        elif text.lower().startswith("/todaypred"):
            try:
                # Build a one-off pre-market snapshot
                price, prev, provider = get_wolf_price()
                summary = _forecast_summary_for_snapshot() or {}
                conf = summary.get("confidence")
                drift = summary.get("drift_daily_pct")
                direction = (
                    "UP"
                    if (isinstance(drift, (int, float)) and float(drift) > 0)
                    else (
                        "DOWN" if (isinstance(drift, (int, float)) and float(drift) < 0) else "FLAT"
                    )
                )
                today = time.strftime("%Y-%m-%d")
                reply = (
                    f"📈 Pre-Market Prediction ({today})\n\n"
                    f"Symbol: {WOLF}\n"
                    f"Prev Close: {('$' + format(prev, '.2f')) if prev is not None else '?'}\n"
                    f"Pre-Market: {('$' + format(price, '.2f')) if price is not None else '?'} ({provider or 'n/a'})\n\n"
                    f"Ghost Forecast: {direction} ({(str(round(float(drift), 2)) + '%') if isinstance(drift, (int, float)) else '?'})\n"
                    f"Confidence: {(str(round(float(conf), 2)) + '%') if isinstance(conf, (int, float)) else '?'}\n"
                    f"Next update: 9:35 AM ET (Open + 5)"
                )
                _tg_send_chat_message(chat_id, reply)
            except Exception as e:
                _tg_send_chat_message(chat_id, f"❌ Error: {str(e)[:150]}")

        elif text.lower().startswith("/buy"):
            # Parse: /buy AAPL 10
            try:
                parts = text.split()
                if len(parts) < 3:
                    _tg_send_chat_message(chat_id, "Usage: /buy SYMBOL QTY\nExample: /buy AAPL 10")
                    return {"ok": True}

                symbol = parts[1].upper()
                qty = float(parts[2])

                from core.alpaca_broker import get_broker
                from core.risk_engine import get_risk_engine

                broker = get_broker()
                risk_engine = get_risk_engine()

                if not broker.enabled:
                    _tg_send_chat_message(chat_id, "⚠️ Broker not enabled")
                    return {"ok": True}

                # Get account info
                account = broker.get_account()
                buying_power = float(account.get("buying_power", 0))

                # Get current price
                try:
                    price, _, _ = get_wolf_price() if symbol == "WOLF" else (None, None, None)
                    if not price:
                        # Try to get price from broker — DO NOT use placeholder
                        price = None
                except Exception:
                    price = None

                if price is None:
                    await _tg_send_chat_message(chat_id, f"❌ Cannot execute /buy — unable to fetch current price for {symbol}. Order blocked for safety.")
                    return {"ok": True}

                # Risk check
                allowed, reason = risk_engine.risk_check_order(
                    {"symbol": symbol, "qty": qty, "side": "buy", "type": "market"},
                    float(account.get("portfolio_value", buying_power)),
                    float(account.get("equity", buying_power)),
                    [],
                )

                if not allowed:
                    _tg_send_chat_message(chat_id, f"🚫 Order blocked by risk engine:\n{reason}")
                    return {"ok": True}

                # Submit order
                order = broker.submit_order(
                    symbol=symbol, qty=qty, side="buy", order_type="market", time_in_force="day"
                )

                if order:
                    order_id = order.get("id", "N/A")
                    _tg_send_chat_message(
                        chat_id,
                        f"✅ BUY order submitted!\n\n"
                        f"Symbol: {symbol}\n"
                        f"Qty: {qty}\n"
                        f"Order ID: {order_id}\n"
                        f"Status: {order.get('status', 'submitted')}",
                    )
                else:
                    _tg_send_chat_message(chat_id, "❌ Order submission failed")
            except ValueError:
                _tg_send_chat_message(chat_id, "❌ Invalid quantity. Must be a number.")
            except Exception as e:
                _tg_send_chat_message(chat_id, f"❌ Error: {str(e)[:100]}")

        elif text.lower().startswith("/sell"):
            # Parse: /sell AAPL
            try:
                parts = text.split()
                if len(parts) < 2:
                    _tg_send_chat_message(chat_id, "Usage: /sell SYMBOL\nExample: /sell AAPL")
                    return {"ok": True}

                symbol = parts[1].upper()

                from core.alpaca_broker import get_broker

                broker = get_broker()

                if not broker.enabled:
                    _tg_send_chat_message(chat_id, "⚠️ Broker not enabled")
                    return {"ok": True}

                # Close position
                result = broker.close_position(symbol)

                if result:
                    order_id = result.get("id", "N/A")
                    _tg_send_chat_message(
                        chat_id,
                        f"✅ SELL order submitted!\n\n"
                        f"Symbol: {symbol}\n"
                        f"Closing entire position\n"
                        f"Order ID: {order_id}",
                    )
                else:
                    _tg_send_chat_message(
                        chat_id, f"❌ No position found for {symbol} or close failed"
                    )
            except Exception as e:
                _tg_send_chat_message(chat_id, f"❌ Error: {str(e)[:100]}")

        elif text.lower().startswith("/watch"):
            # Add crypto to watchlist: /watch BTC
            try:
                parts = text.split()
                if len(parts) < 2:
                    _tg_send_chat_message(chat_id, "Usage: /watch SYMBOL\nExample: /watch BTC")
                    return {"ok": True}

                symbol = parts[1].upper()

                # Add to watchlist
                from core.crypto.crypto_watchlist import add_to_watchlist, get_crypto_watchlist

                added = add_to_watchlist(symbol)
                watchlist = get_crypto_watchlist()

                if added:
                    _tg_send_chat_message(
                        chat_id,
                        f"✅ Added {symbol} to watchlist!\n\n"
                        f"📋 Now tracking {len(watchlist)} cryptos:\n{', '.join(watchlist)}",
                    )
                else:
                    _tg_send_chat_message(chat_id, f"✅ {symbol} already in watchlist")
            except Exception as e:
                _tg_send_chat_message(chat_id, f"❌ Error: {str(e)[:100]}")

        elif text.lower().startswith("/unwatch"):
            # Remove crypto from watchlist: /unwatch BTC
            try:
                parts = text.split()
                if len(parts) < 2:
                    _tg_send_chat_message(chat_id, "Usage: /unwatch SYMBOL\nExample: /unwatch BTC")
                    return {"ok": True}

                symbol = parts[1].upper()

                # Remove from watchlist
                from core.crypto.crypto_watchlist import get_crypto_watchlist, remove_from_watchlist

                removed = remove_from_watchlist(symbol)
                watchlist = get_crypto_watchlist()

                if removed:
                    _tg_send_chat_message(
                        chat_id,
                        f"✅ Removed {symbol} from watchlist\n\n"
                        f"📋 Now tracking {len(watchlist)} cryptos:\n{', '.join(watchlist)}",
                    )
                else:
                    _tg_send_chat_message(chat_id, f"⚠️ {symbol} not in watchlist")
            except Exception as e:
                _tg_send_chat_message(chat_id, f"❌ Error: {str(e)[:100]}")

        elif text.lower().startswith("/cryptos") or text.lower().startswith("/coins"):
            # List all cryptos being tracked
            try:
                from core.crypto.crypto_watchlist import get_crypto_watchlist

                watchlist = get_crypto_watchlist()

                _tg_send_chat_message(
                    chat_id,
                    f"📋 Tracking {len(watchlist)} cryptos:\n\n{', '.join(watchlist)}\n\n"
                    f"Use /watch SYMBOL to add more\n"
                    f"Use /unwatch SYMBOL to remove",
                )
            except Exception as e:
                _tg_send_chat_message(chat_id, f"❌ Error: {str(e)[:100]}")

        elif text.lower().startswith("/predict"):
            # Manually trigger prediction (for testing)
            try:
                if SCHEDULED_PREDICTIONS_ENABLED:
                    _tg_send_chat_message(chat_id, "🔮 Generating prediction now...")
                    scheduled_predictions.force_multi_prediction()
                else:
                    _tg_send_chat_message(chat_id, "⚠️ Prediction scheduler not enabled")
            except Exception as e:
                _tg_send_chat_message(chat_id, f"❌ Error: {str(e)[:100]}")

        elif text.lower().startswith("/check"):
            # Manually check prediction accuracy
            try:
                # Get latest prediction for WOLF from in-memory store
                pred = _LATEST_PREDICTIONS.get("WOLF")
                if not pred:
                    _tg_send_chat_message(chat_id, "No recent prediction to check. Try /predict first.")
                else:
                    # Get current price
                    price, prev, provider = get_wolf_price()
                    pred_price = pred.get("price_at_prediction", prev)
                    pred_direction = pred.get("direction", "FLAT")
                    pred_confidence = pred.get("confidence", 0) * 100

                    # Calculate actual change
                    change_pct = ((price - pred_price) / pred_price * 100) if pred_price else 0
                    actual_direction = "UP" if change_pct > 1 else ("DOWN" if change_pct < -1 else "FLAT")

                    # Determine correctness
                    correct = actual_direction == pred_direction
                    result_emoji = "✅" if correct else "❌"

                    # Format message
                    msg = "⚠️ PREDICTION CHECK\n\n"
                    msg += "PREDICTED:\n"
                    msg += f"  Direction: {pred_direction}\n"
                    msg += f"  Price: ${pred_price:.2f}\n"
                    msg += f"  Confidence: {pred_confidence:.0f}%\n\n"
                    msg += "ACTUAL:\n"
                    msg += f"  Direction: {actual_direction}\n"
                    msg += f"  Price: ${price:.2f} ({change_pct:+.2f}%)\n\n"
                    msg += f"RESULT: {result_emoji} {'CORRECT' if correct else 'INCORRECT'}"

                    _tg_send_chat_message(chat_id, msg)
            except Exception as e:
                _tg_send_chat_message(chat_id, f"❌ Error: {str(e)[:100]}")

        elif text.lower().startswith("/help"):
            reply = (
                "🤖 Ghost AI Commands:\n\n"
                "📊 STOCK TRADING:\n"
                "  /status - Portfolio status\n"
                "  /signal - Current trading signal\n"
                "  /pnl - Daily P&L\n"
                "  /positions - Show open positions\n"
                "  /buy SYMBOL QTY - Buy stocks\n"
                "  /sell SYMBOL - Sell position\n\n"
                "🪙 CRYPTO:\n"
                "  /cryptos - Show watchlist\n"
                "  /watch BTC - Add to watchlist\n"
                "  /unwatch BTC - Remove from watchlist\n\n"
                "🔮 PREDICTIONS:\n"
                "  /predict - Force prediction now\n"
                "  /check - Check prediction accuracy\n\n"
                "📅 Auto-scheduled:\n"
                "  • 8:00 AM ET - Pre-market prediction\n"
                "  • 9:35 AM ET - Market open check\n\n"
                "💬 Ask me anything!\n"
                "Example: 'Should I buy PEPE? 30-day outlook?'"
            )
            _tg_send_chat_message(chat_id, reply)
        elif text.startswith("/"):
            # Unknown command
            _tg_send_chat_message(chat_id, "Unknown command. Try /help")
        else:
            # Natural language question - use AI
            _tg_send_chat_message(chat_id, "🤔 Thinking...")
            answer = _ask_ghost_ai(text)
            _tg_send_chat_message(chat_id, f"🤖 Ghost:\n\n{answer}")
    except Exception as e:
        LOGGER.error(f"Telegram webhook error: {e}", exc_info=True)
        _tg_send_chat_message(chat_id, "❌ Error processing message.")
    return {"ok": True}


@router.post("/api/alerts/hold")
async def api_alerts_hold(
    t: AlertToggle, credentials: HTTPAuthorizationCredentials | None = AUTH_DEP
):
    # Optional bearer (if token set)
    _require_bearer(
        (f"Bearer {credentials.credentials}") if credentials and credentials.credentials else None
    )
    ALERT_STATE["hold_override"] = bool(t.hold)
    _set_hold_gauge()
    return {"ok": True, "hold_override": ALERT_STATE["hold_override"]}


@router.get("/api/alerts/config")
async def api_alerts_config_get():
    return {
        "mode": ALERT_MODE,
        "buy_pct": ALERT_BUY_PCT,
        "sell_pct": ALERT_SELL_PCT,
        "band_pct": BAND_PCT,
        "trail_sell_pct": TRAIL_SELL_PCT,
        "trail_buy_pct": TRAIL_BUY_PCT,
        "throttle_s": ALERT_THROTTLE_S,
        "throttle_buy_s": ALERT_THROTTLE_BUY_S,
        "throttle_sell_s": ALERT_THROTTLE_SELL_S,
        "vol_gate": VOL_GATE,
        "vol_lookback_days": VOL_LOOKBACK_DAYS,
        "vol_k": VOL_K,
        "vol_ttl_s": VOL_TTL_S,
        "trailing_high": ALERT_STATE.get("trailing_high"),
        "trailing_low": ALERT_STATE.get("trailing_low"),
        "schedule_open_close": bool(SCHEDULE_OPEN_CLOSE),
        "schedule_window_s": SCHEDULE_WINDOW_S,
    }


@router.post("/api/alerts/config")
async def api_alerts_config_post(
    body: AlertConfigBody, credentials: HTTPAuthorizationCredentials | None = AUTH_DEP
):
    _require_bearer(
        (f"Bearer {credentials.credentials}") if credentials and credentials.credentials else None
    )
    global ALERT_MODE, ALERT_BUY_PCT, ALERT_SELL_PCT, BAND_PCT, TRAIL_SELL_PCT, TRAIL_BUY_PCT
    global ALERT_THROTTLE_S, ALERT_THROTTLE_BUY_S, ALERT_THROTTLE_SELL_S
    global VOL_GATE, VOL_LOOKBACK_DAYS, VOL_K, VOL_TTL_S
    global SCHEDULE_OPEN_CLOSE, SCHEDULE_WINDOW_S
    # Validate and apply
    if body.mode is not None:
        m = body.mode.strip().lower()
        if m not in ("fixed", "band", "trailing"):
            raise HTTPException(422, "mode must be one of: fixed, band, trailing")
        ALERT_MODE = m
        # Reset trailing anchors when switching modes
        if m != "trailing":
            ALERT_STATE["trailing_high"] = None
            ALERT_STATE["trailing_low"] = None
    if body.buy_pct is not None:
        if not (0 < body.buy_pct < 1):
            raise HTTPException(422, "buy_pct should be in (0,1)")
        ALERT_BUY_PCT = float(body.buy_pct)
    if body.sell_pct is not None:
        if not (1 < body.sell_pct < 2):
            raise HTTPException(422, "sell_pct should be in (1,2)")
        ALERT_SELL_PCT = float(body.sell_pct)
    if body.band_pct is not None:
        if not (0 < body.band_pct < 1):
            raise HTTPException(422, "band_pct should be in (0,1)")
        BAND_PCT = float(body.band_pct)
    if body.trail_sell_pct is not None:
        if not (0 < body.trail_sell_pct < 1):
            raise HTTPException(422, "trail_sell_pct should be in (0,1)")
        TRAIL_SELL_PCT = float(body.trail_sell_pct)
    if body.trail_buy_pct is not None:
        if not (0 < body.trail_buy_pct < 1):
            raise HTTPException(422, "trail_buy_pct should be in (0,1)")
        TRAIL_BUY_PCT = float(body.trail_buy_pct)
    if body.throttle_s is not None:
        if body.throttle_s < 0:
            raise HTTPException(422, "throttle_s must be >= 0")
        ALERT_THROTTLE_S = int(body.throttle_s)
    if body.throttle_buy_s is not None:
        if body.throttle_buy_s < 0:
            raise HTTPException(422, "throttle_buy_s must be >= 0")
        ALERT_THROTTLE_BUY_S = int(body.throttle_buy_s)
    if body.throttle_sell_s is not None:
        if body.throttle_sell_s < 0:
            raise HTTPException(422, "throttle_sell_s must be >= 0")
        ALERT_THROTTLE_SELL_S = int(body.throttle_sell_s)
    if body.vol_gate is not None:
        VOL_GATE = 1 if int(body.vol_gate) else 0
    if body.vol_lookback_days is not None:
        if body.vol_lookback_days <= 0:
            raise HTTPException(422, "vol_lookback_days must be > 0")
        VOL_LOOKBACK_DAYS = int(body.vol_lookback_days)
        ALERT_STATE["vol_ts"] = 0.0  # drop cache
    if body.vol_k is not None:
        if body.vol_k <= 0:
            raise HTTPException(422, "vol_k must be > 0")
        VOL_K = float(body.vol_k)
    if body.vol_ttl_s is not None:
        if body.vol_ttl_s <= 0:
            raise HTTPException(422, "vol_ttl_s must be > 0")
        VOL_TTL_S = int(body.vol_ttl_s)
        ALERT_STATE["vol_ts"] = 0.0
    if body.schedule_open_close is not None:
        SCHEDULE_OPEN_CLOSE = 1 if int(body.schedule_open_close) else 0
        # start/stop worker accordingly
        try:
            if SCHEDULE_OPEN_CLOSE:
                _start_schedule_worker()
            else:
                _stop_schedule_worker()
        except Exception:
            pass
    if body.schedule_window_s is not None:
        if body.schedule_window_s <= 0:
            raise HTTPException(422, "schedule_window_s must be > 0")
        SCHEDULE_WINDOW_S = int(body.schedule_window_s)
    _set_mode_gauge()
    _set_hold_gauge()
    return await api_alerts_config_get()


@router.post("/api/alerts/dispatch")
async def api_alerts_dispatch(
    dry_run: int = 0,
    credentials: HTTPAuthorizationCredentials | None = AUTH_DEP,
    idempotency_key: str | None = Header(
        default=None, convert_underscores=False, alias="Idempotency-Key"
    ),
):
    _require_bearer(
        (f"Bearer {credentials.credentials}") if credentials and credentials.credentials else None
    )
    # purge expired idempotency
    try:
        now_ts = time.time()
        for k, ts in _IDEMP_CACHE_TS.items():
            if now_ts - ts > _IDEMPOTENCY_TTL_S:
                _IDEMP_CACHE_TS.pop(k, None)
                _IDEMP_CACHE.pop(k, None)
    except Exception:
        pass
    if idempotency_key:
        prior = _IDEMP_CACHE.get(idempotency_key)
        if prior is not None:
            return prior
    sig = _evaluate_signal()
    now = time.time()
    # throttle duplicates
    if (
        not dry_run
        and ALERT_STATE.get("last_signal") == sig
        and (now - ALERT_STATE.get("last_sent_ts", 0)) < ALERT_THROTTLE_S
    ):
        try:
            if _C_ALERT_THROTTLED is not None:
                _C_ALERT_THROTTLED.inc()
        except Exception:
            pass
        return {"ok": True, "throttled": True, "reason": "duplicate", "signal": sig}
    # per-action cooldown window
    window = ALERT_THROTTLE_S
    last_ts = float(ALERT_STATE.get("last_sent_ts", 0.0))
    act = str(sig.get("action") or "HOLD").upper()
    if act == "BUY":
        window = ALERT_THROTTLE_BUY_S
        last_ts = float(ALERT_STATE.get("last_sent_ts_buy", 0.0))
    elif act == "SELL":
        window = ALERT_THROTTLE_SELL_S
        last_ts = float(ALERT_STATE.get("last_sent_ts_sell", 0.0))
    if not dry_run and (now - last_ts) < window:
        try:
            if _C_ALERT_THROTTLED is not None:
                _C_ALERT_THROTTLED.inc()
        except Exception:
            pass
        return {"ok": True, "throttled": True, "reason": "cooldown", "signal": sig}
    # send
    if dry_run:
        ok = True
        result_label = "dry-run"
    else:
        text = _signal_card(sig)
        enq_ok = enqueue_alert_text(text, sig)
        result_label = "queued" if enq_ok else "queue-fail"
        if enq_ok:
            ALERT_STATE["last_signal"] = sig
            ALERT_STATE["last_sent_ts"] = now
            if act == "BUY":
                ALERT_STATE["last_sent_ts_buy"] = now
            elif act == "SELL":
                ALERT_STATE["last_sent_ts_sell"] = now
    try:
        if _C_ALERT_SENT is not None:
            _C_ALERT_SENT.labels(
                action=sig.get("action") or "?",
                mode=sig.get("mode") or "?",
                result=result_label,
            ).inc()
    except Exception:
        pass
    resp = {
        "ok": bool(ok if dry_run else (result_label == "queued")),
        "signal": sig,
        "dry_run": bool(dry_run),
        "queued": (result_label == "queued"),
    }
    if idempotency_key:
        _IDEMP_CACHE[idempotency_key] = resp
        _IDEMP_CACHE_TS[idempotency_key] = time.time()
    try:
        _add_event(
            "alerts.dispatch",
            "Alert dispatch",
            {
                "result": result_label,
                "dry_run": bool(dry_run),
                "action": sig.get("action"),
            },
        )
    except Exception:
        pass
    return resp


@router.post("/alerts/status")
async def alerts_status(credentials: HTTPAuthorizationCredentials | None = AUTH_DEP):
    _require_bearer(
        (f"Bearer {credentials.credentials}") if credentials and credentials.credentials else None
    )
    global _STATUS_LAST_TS, _STATUS_LAST_HASH
    q = float(STATE.get("qty", 0.0))
    a = float(STATE.get("avg_cost", 0.0))
    price, prev, provider = get_wolf_price()
    rid = _cv_trace_id.get()
    # Pre-compute derived values for safe formatting
    current = price if price is not None else a
    market_value = float(q * current) if current else 0.0
    if a > 0:
        pnl_abs = (current - a) * q
        pnl_pct = ((current - a) / a) * 100.0
    else:
        pnl_abs = 0.0
        pnl_pct = 0.0
    ctx = {
        "symbol": "WOLF",
        "name": "Wolfspeed",
        "qty": f"{q:.8f}",
        "avg": f"${a:.2f}",
        "price": ("?" if price is None else f"${price:.2f}"),
        "provider": provider or "unavailable",
        "request_id": rid if rid and rid != "-" else "",
        "market_value": market_value,
        "pnl_abs": pnl_abs,
        "pnl_pct": pnl_pct,
    }
    tpl = ALERT_CONFIG.get("status_template")
    if tpl:
        card = _render_template(tpl, ctx)
    else:
        card = (
            "📊 STATUS — WOLF (Wolfspeed)\n"
            f"• Qty: {ctx['qty']}\n"
            f"• Avg Cost: {ctx['avg']}\n"
            f"• Price: {ctx['price']} ({ctx['provider']})\n"
            f"• Market Value: ${ctx['market_value']:.2f}\n"
            f"• PnL: {ctx['pnl_abs']:.2f} ({ctx['pnl_pct']:.6f}%)\n"
            + (f"\nReq: {ctx['request_id']}" if ctx["request_id"] else "")
        )
    # Merge-guard: dedupe identical payloads for a short window
    try:
        card_hash = hashlib.sha256(card.encode("utf-8")).hexdigest()
        now = time.time()
        if _STATUS_LAST_HASH == card_hash and (now - _STATUS_LAST_TS) < STATUS_MERGE_TTL_S:
            return {"ok": True, "throttled": True, "reason": "merge"}
        if (now - _STATUS_LAST_TS) < STATUS_THROTTLE_S:
            return {"ok": True, "throttled": True, "reason": "cooldown"}
        _STATUS_LAST_HASH = card_hash
        _STATUS_LAST_TS = now
    except Exception:
        pass
    ok = enqueue_alert_text(card)
    return {"ok": bool(ok), "price": price, "provider": provider}


@router.get("/alerts/status/preview")
async def alerts_status_preview():
    price, prev, provider = get_wolf_price()
    text = _build_status_card(price=price, provider=provider, include_req=False)
    return {"text": text}


@router.post("/api/alerts/test")
async def api_alerts_test():
    """Send test Telegram alert to validate configuration."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return JSONResponse(
            {"ok": False, "detail": "Telegram not configured (missing BOT_TOKEN or CHAT_ID)"},
            status_code=503
        )

    try:
        # Format timestamp in America/Chicago timezone
        try:
            if ZoneInfo:
                tz_ct = ZoneInfo("America/Chicago")
                dt_ct = datetime.now(tz_ct)
                ts_ct = dt_ct.strftime("%Y-%m-%d %I:%M %p CT")
            else:
                ts_ct = datetime.now().strftime("%Y-%m-%d %I:%M %p UTC")
        except Exception:
            ts_ct = datetime.now().strftime("%Y-%m-%d %I:%M %p UTC")

        test_message = f"🤖 Ghost alert test: {ts_ct} | OK"

        # Use existing Telegram send function
        ok, results = send_telegram_detailed(test_message)

        if ok and results:
            # Extract message_id from first successful result
            message_id = None
            for res in results:
                if res.get("ok") and res.get("response"):
                    try:
                        message_id = res["response"].get("result", {}).get("message_id")
                        if message_id:
                            break
                    except Exception:
                        pass

            return {
                "ok": True,
                "message": "Test alert sent successfully",
                "message_id": message_id,
                "ts": int(time.time() * 1000)
            }
        else:
            return JSONResponse(
                {"ok": False, "detail": "Telegram send failed", "results": results},
                status_code=500
            )
    except Exception as e:
        LOGGER.error(f"alerts_test_error: {e}")
        return JSONResponse(
            {"ok": False, "detail": str(e)},
            status_code=500
        )


@router.get("/api/recent_alerts")
async def api_recent_alerts(limit: int = 10):
    """
    Get recent Cash-App style alerts from last 24h.

    Query params:
        limit: Max alerts to return (default 10)

    Returns:
        {
            'ok': True,
            'alerts': [
                {
                    'symbol': 'WEPE',
                    'message': 'WEPE +12.5% (1h)\\nPrice: $0.000123\\nVolume: 3x surge',
                    'timestamp': 1731654000,
                    'tier': 'VIP'
                },
                ...
            ],
            'count': 10
        }
    """
    try:
        from core.telegram_alerts import get_recent_alerts

        alerts = get_recent_alerts(limit=limit)

        return {
            'ok': True,
            'alerts': alerts,
            'count': len(alerts),
            'timestamp': int(time.time())
        }
    except Exception as e:
        LOGGER.error(f"Recent alerts API failed: {e}")
        return {
            'ok': False,
            'error': str(e),
            'alerts': [],
            'count': 0,
            'timestamp': int(time.time())
        }


