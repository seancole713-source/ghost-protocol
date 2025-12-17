"""
Demo API Endpoints
==================

Quick endpoints to trigger demo features instantly.
"""

from fastapi import APIRouter, HTTPException
from core.guardian_heartbeat_scheduler import GuardianHeartbeatScheduler
import logging

router = APIRouter(prefix="/api/demo", tags=["demo"])
logger = logging.getLogger(__name__)


@router.post("/morning_now")
async def trigger_morning_now():
    """
    🎬 DEMO: Send morning prophecy to Telegram RIGHT NOW
    
    This triggers an immediate morning prophecy (doesn't wait for 6 AM).
    Perfect for testing the $200/day format.
    
    Requires:
    - GHOST_DEMO_MODE=1 (for positive predictions)
    - TELEGRAM_BOT_TOKEN set
    - TELEGRAM_CHAT_ID set
    
    Returns:
        {"ok": true, "message": "Morning prophecy sent!"}
    """
    try:
        logger.info("🎬 Demo: Triggering morning prophecy NOW")
        
        # Import here to avoid circular dependencies
        from core.telegram_hunter import send_telegram_message
        
        # Send test message first
        test_sent = send_telegram_message("🎬 Generating your $200/day investment plan...")
        logger.info(f"Test message sent: {test_sent}")
        
        # Generate and send prophecy
        scheduler = GuardianHeartbeatScheduler()
        await scheduler._send_heartbeat('morning', position_size=100.0)
        
        return {
            "ok": True,
            "message": "Morning prophecy sent to Telegram!",
            "position_size": 100.0,
            "demo_mode": True,
            "test_sent": test_sent
        }
        
    except Exception as e:
        logger.exception(f"Failed to send morning prophecy: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def demo_status():
    """
    Check demo mode status
    
    Returns current demo configuration.
    """
    import os
    
    return {
        "demo_mode": os.environ.get("GHOST_DEMO_MODE", "0") == "1",
        "telegram_configured": bool(
            os.environ.get("TELEGRAM_BOT_TOKEN") and 
            os.environ.get("TELEGRAM_CHAT_ID")
        ),
        "telegram_chat_id": os.environ.get("TELEGRAM_CHAT_ID", "not_set"),
        "position_size": 100.0,
        "ready": True
    }
