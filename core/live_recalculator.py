"""
🔄 LIVE RECALCULATOR
Real-time confidence & target updates for active daily picks
Runs every 5 minutes during market hours
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Any, Dict
from zoneinfo import ZoneInfo

LOGGER = logging.getLogger(__name__)

CHICAGO_TZ = ZoneInfo("America/Chicago")

# Active positions tracking
_ACTIVE_PICKS: dict[str, dict[str, Any]] = {}
_LAST_UPDATE: dict[str, float] = {}
_UPDATE_INTERVAL = 300  # 5 minutes


# ============================================================================
# LIVE RECALCULATION ENGINE
# ============================================================================

async def recalculate_pick(pick: dict[str, Any]) -> dict[str, Any]:
    """
    Recalculate confidence, expected gain, and price targets for active pick
    """
    symbol = pick["symbol"]
    
    try:
        # Get fresh market data
        from core.providers.turbo_provider import get_turbo_provider
        turbo = get_turbo_provider()
        
        data = await turbo.get_price_async(symbol)
        if not data:
            return None
        
        current_price = data.get("price", 0)
        original_entry = pick["prices"]["entry_low"]
        
        # Calculate current P&L
        pnl_pct = ((current_price - original_entry) / original_entry) * 100
        
        # Recalculate confidence scores
        from core.daily_predictions_engine import calculate_confidence_score
        score_data = await calculate_confidence_score(symbol, pick["asset_type"])
        new_confidence = score_data["confidence"]
        
        # Calculate new expected gain based on current momentum
        time_elapsed_min = (time.time() - pick.get("entry_time", time.time())) / 60
        price_velocity = pnl_pct / max(1, time_elapsed_min)  # %/minute
        
        # Project to 24hr
        minutes_remaining = 1440 - time_elapsed_min
        projected_gain = pnl_pct + (price_velocity * minutes_remaining)
        
        # Adjust by new confidence
        new_expected_gain = projected_gain * (new_confidence / 100)
        
        # Recalculate price targets
        volatility = data.get("volatility", 0.02)
        
        new_target = current_price * (1 + new_expected_gain * 0.75 / 100)
        new_peak = current_price * (1 + new_expected_gain * 1.3 / 100)
        
        # Trail stop logic
        if pnl_pct > 10:  # If up >10%, trail stop
            trail_stop = current_price * 0.92  # 8% below current
        else:
            trail_stop = original_entry * 0.94  # 6% below entry
        
        # Determine action
        confidence_change = new_confidence - pick["confidence"]
        gain_change = new_expected_gain - pick["expected_gain"]
        
        if new_confidence < 50 or new_expected_gain < 0:
            action = "EXIT"
            reason = "Confidence dropped below 50% or negative expected gain"
        elif confidence_change > 15 and gain_change > 10:
            action = "ADD"
            reason = "Momentum strengthening significantly"
        elif pnl_pct >= (pick["expected_gain"] * 0.9):  # Near target
            action = "TAKE_PROFITS"
            reason = "Approaching profit target"
        elif current_price <= trail_stop:
            action = "STOP_HIT"
            reason = "Trail stop triggered"
        else:
            action = "HOLD"
            reason = "On track"
        
        return {
            "symbol": symbol,
            "current_price": round(current_price, 2),
            "pnl_pct": round(pnl_pct, 2),
            "new_confidence": round(new_confidence, 1),
            "new_expected_gain": round(new_expected_gain, 1),
            "new_target": round(new_target, 2),
            "new_peak": round(new_peak, 2),
            "trail_stop": round(trail_stop, 2),
            "action": action,
            "reason": reason,
            "confidence_change": round(confidence_change, 1),
            "gain_change": round(new_expected_gain - pick["expected_gain"], 1),
            "score_breakdown": score_data,
            "time_in_trade": round(time_elapsed_min / 60, 1)  # hours
        }
        
    except Exception as e:
        LOGGER.error(f"Recalculation failed for {symbol}: {e}")
        return None


# ============================================================================
# POSITION MONITORING
# ============================================================================

async def monitor_active_picks():
    """
    Monitor all active picks, recalculate scores, send alerts
    """
    if not _ACTIVE_PICKS:
        return
    
    LOGGER.info(f"📊 Monitoring {len(_ACTIVE_PICKS)} active picks...")
    
    for symbol, pick in list(_ACTIVE_PICKS.items()):
        try:
            # Check if enough time passed since last update
            last_update = _LAST_UPDATE.get(symbol, 0)
            if time.time() - last_update < _UPDATE_INTERVAL:
                continue
            
            # Recalculate
            updated = await recalculate_pick(pick)
            
            if not updated:
                continue
            
            # Check if significant change (needs alert)
            needs_alert = False
            
            if updated["action"] in ["EXIT", "ADD", "TAKE_PROFITS", "STOP_HIT"]:
                needs_alert = True
            elif abs(updated["confidence_change"]) > 15:  # >15% confidence change
                needs_alert = True
            elif abs(updated["gain_change"]) > 10:  # >10% gain expectation change
                needs_alert = True
            
            if needs_alert:
                await send_live_update_alert(pick, updated)
            
            # Update tracking
            _LAST_UPDATE[symbol] = time.time()
            
            # If exit signal, remove from active tracking
            if updated["action"] in ["EXIT", "STOP_HIT", "TAKE_PROFITS"]:
                del _ACTIVE_PICKS[symbol]
                LOGGER.info(f"🔴 {symbol} removed from active tracking ({updated['action']})")
            
        except Exception as e:
            LOGGER.error(f"Failed to monitor {symbol}: {e}")


async def send_live_update_alert(original_pick: dict[str, Any], updated: dict[str, Any]):
    """
    Send formatted live update to Telegram
    """
    try:
        from core.alert_manager import send_live_update
        await send_live_update(original_pick, updated)
    except Exception as e:
        LOGGER.error(f"Failed to send live update: {e}")


# ============================================================================
# PICK MANAGEMENT
# ============================================================================

def add_active_pick(pick: dict[str, Any]):
    """
    Add new pick to active monitoring
    """
    symbol = pick["symbol"]
    pick["entry_time"] = time.time()
    _ACTIVE_PICKS[symbol] = pick
    _LAST_UPDATE[symbol] = time.time()
    LOGGER.info(f"✅ {symbol} added to active tracking")


def remove_active_pick(symbol: str):
    """
    Remove pick from active monitoring
    """
    if symbol in _ACTIVE_PICKS:
        del _ACTIVE_PICKS[symbol]
    if symbol in _LAST_UPDATE:
        del _LAST_UPDATE[symbol]
    LOGGER.info(f"🔴 {symbol} removed from active tracking")


def get_active_picks() -> list[dict[str, Any]]:
    """
    Get list of all active picks
    """
    return list(_ACTIVE_PICKS.values())


# ============================================================================
# BACKGROUND MONITORING LOOP
# ============================================================================

async def live_recalculator_loop():
    """
    Continuous loop: Monitor active picks every 5 minutes
    """
    LOGGER.info("🔄 Live recalculator started")
    
    while True:
        try:
            now = datetime.now(CHICAGO_TZ)
            
            # Only run during market hours (9:30 AM - 4:00 PM CT)
            if 9 <= now.hour < 16:
                await monitor_active_picks()
                await asyncio.sleep(300)  # 5 minutes
            elif 16 <= now.hour < 20:  # After hours crypto monitoring
                await monitor_active_picks()
                await asyncio.sleep(600)  # 10 minutes
            else:
                # Overnight: check less frequently
                await asyncio.sleep(1800)  # 30 minutes
                
        except Exception as e:
            LOGGER.error(f"Live recalculator error: {e}", exc_info=True)
            await asyncio.sleep(300)
