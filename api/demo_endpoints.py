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
        from core.daily_top_10_scanner import DailyTop10Scanner
        from core.guardian_oracle import get_guardian_oracle
        
        # Send test message first
        test_sent = send_telegram_message("🎬 Generating your $200/day investment plan...")
        logger.info(f"Test message sent: {test_sent}")
        
        # Run scan directly with logging
        logger.info("Starting scan...")
        scanner = DailyTop10Scanner()
        top_10 = await scanner.scan_for_top_10()
        logger.info(f"Found {len(top_10)} opportunities")
        
        if not top_10:
            message = "🔮 No opportunities found meeting criteria today."
            send_telegram_message(message)
            return {
                "ok": True,
                "message": "No opportunities found",
                "position_size": 100.0,
                "test_sent": test_sent,
                "demo_mode": True
            }
        
        # Save to database
        scanner.save_top_10(top_10)
        logger.info("Saved to database")
        
        # Format prophecy
        logger.info("Formatting prophecy...")
        guardian = get_guardian_oracle()
        prophecy = await guardian.morning_prophecy(top_10, position_size=100.0)
        logger.info(f"Prophecy length: {len(prophecy)} chars")
        
        # Split if too long (Telegram limit 4096)
        if len(prophecy) > 4000:
            logger.warning(f"Message too long ({len(prophecy)} chars), splitting...")
            parts = []
            lines = prophecy.split('\n')
            current_part = []
            current_length = 0
            
            for line in lines:
                if current_length + len(line) + 1 > 4000:
                    parts.append('\n'.join(current_part))
                    current_part = [line]
                    current_length = len(line)
                else:
                    current_part.append(line)
                    current_length += len(line) + 1
            
            if current_part:
                parts.append('\n'.join(current_part))
            
            logger.info(f"Sending {len(parts)} parts...")
            for i, part in enumerate(parts):
                send_telegram_message(part)
                logger.info(f"Sent part {i+1}/{len(parts)}")
        else:
            send_telegram_message(prophecy)
            logger.info("Prophecy sent!")
        
        return {
            "ok": True,
            "message": "Morning prophecy sent to Telegram!",
            "position_size": 100.0,
            "opportunities_found": len(top_10),
            "prophecy_length": len(prophecy),
            "test_sent": test_sent,
            "demo_mode": True
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
